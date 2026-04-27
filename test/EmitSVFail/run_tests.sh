#!/usr/bin/env bash
# Synthesizability-gate fail-fast tests for -emit-systemverilog.
#
# Each .m here describes a MATLAB construct that has no synthesizable
# RTL form. `matlabc -emit-systemverilog FILE` is expected to:
#
#   - exit with a non-zero status, AND
#   - emit a stderr line that contains the matching .stderr file's
#     content as a substring.
#
# Each .stderr is one short snippet (e.g. "while-loop is not
# synthesizable") so the contract is robust against the exact
# wording of the surrounding diagnostic. We additionally run
# `-check-synthesizable` to confirm it produces the same diagnostic
# without writing any output (the gate-only mode).
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stderr"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .stderr)"; continue; }

  expected="$(cat "$exp")"
  ok=1

  # 1. -emit-systemverilog rejects.
  out=$(mktemp -t emsv.XXXXXX); err=$(mktemp -t emsv.XXXXXX)
  "$MATLABC" -emit-systemverilog "$m" >"$out" 2>"$err"; rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "FAIL $base [emit]: matlabc exited 0 (expected non-zero)"; ok=0
  elif ! grep -qF -- "$expected" "$err"; then
    echo "FAIL $base [emit]: expected stderr substring not found"
    echo "  expected: $expected"
    echo "  stderr:"
    sed 's/^/    /' "$err"
    ok=0
  fi
  rm -f "$out" "$err"

  # 2. -check-synthesizable rejects (gate-only mode).
  out=$(mktemp -t emsv.XXXXXX); err=$(mktemp -t emsv.XXXXXX)
  "$MATLABC" -check-synthesizable "$m" >"$out" 2>"$err"; rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "FAIL $base [check]: -check-synthesizable exited 0"; ok=0
  elif ! grep -qF -- "$expected" "$err"; then
    echo "FAIL $base [check]: expected stderr substring not found"
    echo "  expected: $expected"
    echo "  stderr:"
    sed 's/^/    /' "$err"
    ok=0
  elif [[ -s "$out" ]]; then
    echo "FAIL $base [check]: -check-synthesizable wrote stdout (expected empty)"
    sed 's/^/    /' "$out"
    ok=0
  fi
  rm -f "$out" "$err"

  if [[ $ok -eq 1 ]]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done

echo "----"
echo "emit-sv-fail passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
