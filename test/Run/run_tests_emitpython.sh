#!/usr/bin/env bash
# Build-and-run tests for the Python emission path. For each .m in this
# directory, runs `matlabc -emit-python` and executes the emitted source
# with python3 (adding runtime/ to PYTHONPATH so `matlab_runtime` resolves).
# Compares stdout against the matching .stdout file (or .stdout-python
# override — see below). Programs that are known to diverge (e.g.
# MATLAB-bit-exact numerics) can be marked with a .skip-emit-python
# companion file and are reported as SKIPs.
#
# Per-test Python-specific golden: matrix display via numpy's `print(M)`
# uses bracket / dotted-float formatting that diverges from MATLAB's
# right-aligned `%7g` columns shared with the C/C++ backends. Tests that
# emit matrices can ship a `<name>.stdout-python` override which is
# preferred over `<name>.stdout` for the Python lane only.
#
# Set UPDATE=1 to (re)generate `.stdout-python` from the current emit
# whenever it diverges from `.stdout`. Useful after intentional emitter
# changes.
#
# Usage: run_tests_emitpython.sh <path-to-matlabc>
# Env:   PYTHON=python3  python interpreter (default: python3)
#        UPDATE=1        write .stdout-python on diff (off by default)
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

PY="${PYTHON:-python3}"
UPDATE="${UPDATE:-0}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

pass=0; fail=0; skip=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp_py="${m%.m}.stdout-python"
  exp_default="${m%.m}.stdout"
  if [[ -e "$exp_py" ]]; then
    exp="$exp_py"
  elif [[ -e "$exp_default" ]]; then
    exp="$exp_default"
  else
    echo "SKIP $base (no .stdout)"; continue
  fi

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
      if [[ "$UPDATE" == "1" ]]; then
        printf '%s\n' "$got" > "$exp_py"
        echo "UPDATE $base: wrote $(basename "$exp_py")"
        pass=$((pass+1))
      else
        fail=$((fail+1))
        echo "FAIL $base: stdout mismatch (sorted)"
      fi
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    if [[ "$UPDATE" == "1" ]]; then
      printf '%s\n' "$got" > "$exp_py"
      echo "UPDATE $base: wrote $(basename "$exp_py")"
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch"
    fi
  fi
  rm -f "$tmpsrc"
done

echo "----"
echo "emit-python passed: $pass    failed: $fail    skipped: $skip"
exit $(( fail > 0 ? 1 : 0 ))
