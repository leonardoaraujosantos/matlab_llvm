#!/usr/bin/env bash
# Differential compile-vs-interpret tests.
#
# For each `*.m` in this directory, runs the program TWICE:
#   1. through the interpreter   (`matlabc -repl < file.m`)
#   2. through the compiled lane  (`matlabc -emit-cpp` → c++ → run)
# and diffs the two stdout streams. A mismatch (or a compile/link/run
# failure) is a test failure.
#
# Unlike test/EmitC (which diffs emitted *source* against a golden) and
# test/Run (which diffs compiled output against a hand-authored .stdout),
# this harness needs NO golden: the interpreter IS the reference. That makes
# it cheap to add a fixture for any "normal" language construct, which is the
# coverage that let loop-carried matrix accumulation (`X = X + dt*k`),
# nested matrix expressions, and the unrolled loop counter ship broken.
#
# Usage: run_tests.sh <path-to-matlabc> [path-to-libMatlabRuntime.a]
#   env: CXX=c++  CXXSTD=-std=c++20
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc> [libMatlabRuntime.a]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RTLIB="${2:-$ROOT/build/libMatlabRuntime.a}"
if [[ ! -f "$RTLIB" ]]; then
  echo "error: runtime archive not found: $RTLIB" >&2
  echo "       build the 'MatlabRuntime' target first." >&2
  exit 2
fi
CXX="${CXX:-c++}"
CXXSTD="${CXXSTD:--std=c++20}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Strip trailing blank lines (the REPL emits a few; the compiled binary does
# not) so the comparison is on meaningful output only.
strip_trailing_blanks() { awk 'NF{p=NR} {a[NR]=$0} END{for(i=1;i<=p;i++) print a[i]}' "$1"; }

pass=0; fail=0; failed_names=()
for m in "$DIR"/*.m; do
  [[ -e "$m" ]] || continue
  name="$(basename "$m" .m)"

  # 1. Interpreter reference (strip the one-line REPL banner).
  ref="$TMP/$name.ref"
  if ! "$MATLABC" -repl < "$m" 2>/dev/null | grep -v '^matlabc REPL' > "$ref"; then
    echo "FAIL  $name  (interpreter run failed)"; fail=$((fail+1)); failed_names+=("$name"); continue
  fi

  # 2. Compiled lane: emit C++, compile against the runtime archive, run.
  src="$TMP/$name.cpp"; bin="$TMP/$name.bin"; got="$TMP/$name.got"
  if ! "$MATLABC" -emit-cpp "$m" > "$src" 2>"$TMP/$name.emit.err"; then
    echo "FAIL  $name  (emit-cpp failed)"; sed 's/^/        /' "$TMP/$name.emit.err" | head -3
    fail=$((fail+1)); failed_names+=("$name"); continue
  fi
  if ! "$CXX" "$CXXSTD" -w -I "$ROOT/runtime" -I "$ROOT" "$src" "$RTLIB" -lm -o "$bin" 2>"$TMP/$name.cc.err"; then
    echo "FAIL  $name  (compile/link failed)"; sed 's/^/        /' "$TMP/$name.cc.err" | head -3
    fail=$((fail+1)); failed_names+=("$name"); continue
  fi
  if ! "$bin" > "$got" 2>/dev/null; then
    echo "FAIL  $name  (compiled binary crashed)"; fail=$((fail+1)); failed_names+=("$name"); continue
  fi

  # 3. Compare (trailing blank lines normalized).
  strip_trailing_blanks "$ref" > "$ref.n"
  strip_trailing_blanks "$got" > "$got.n"
  if diff -q "$ref.n" "$got.n" >/dev/null; then
    pass=$((pass+1))
  else
    echo "FAIL  $name  (interpreter vs compiled mismatch)"
    diff "$ref.n" "$got.n" | head -8 | sed 's/^/        /'
    fail=$((fail+1)); failed_names+=("$name")
  fi
done

echo "----"
echo "differential passed: $pass    failed: $fail"
if (( fail > 0 )); then
  echo "failing: ${failed_names[*]}"
  exit 1
fi
