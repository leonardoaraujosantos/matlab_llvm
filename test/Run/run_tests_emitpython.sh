#!/usr/bin/env bash
# Build-and-run tests for the Python emission path. For each .m in this
# directory, runs `matlabc -emit-python` and executes the emitted source
# with python3 (adding runtime/ to PYTHONPATH so `matlab_runtime` resolves).
# Compares stdout against the matching .stdout file. Programs that are
# known to diverge (e.g. MATLAB-bit-exact numerics) can be marked with a
# .skip-emit-python companion file and are reported as SKIPs.
#
# Usage: run_tests_emitpython.sh <path-to-matlabc>
# Env:   PYTHON=python3  python interpreter (default: python3)
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

PY="${PYTHON:-python3}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

pass=0; fail=0; skip=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .stdout)"; continue; }

  if [[ -e "${m%.m}.skip-emit-python" ]]; then
    echo "SKIP $base (marked .skip-emit-python)"
    skip=$((skip+1)); continue
  fi

  tmpsrc="$(mktemp -t mlp.XXXXXX).py"

  if ! "$MATLABC" -emit-python "$m" > "$tmpsrc" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-python errored"
    fail=$((fail+1))
    rm -f "$tmpsrc"; continue
  fi

  got="$(PYTHONPATH="$RUNTIME_DIR" "$PY" "$tmpsrc" 2>/dev/null)" || {
    echo "FAIL $base: python3 non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpsrc"; continue
  }

  if [[ -e "${m%.m}.sorted" ]]; then
    if diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch (sorted)"
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
  fi
  rm -f "$tmpsrc"
done

echo "----"
echo "emit-python passed: $pass    failed: $fail    skipped: $skip"
exit $(( fail > 0 ? 1 : 0 ))
