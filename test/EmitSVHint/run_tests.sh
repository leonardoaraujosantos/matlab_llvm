#!/usr/bin/env bash
# Lint-hint test suite for -emit-systemverilog (Workstream D).
#
# Each fixture pairs a `.m` with a `.stderr` file describing
# expected emitter warnings. The runner asserts:
#   - matlabc -emit-systemverilog FILE exits 0 (warnings are
#     informational; emission still succeeds).
#   - stderr CONTAINS each non-blank line of the .stderr as a
#     substring (or stderr is empty when .stderr is empty —
#     control-case fixtures that should not trigger a hint).
#
# Hints are stderr-only; goldens for the SV body live alongside
# the EmitSV golden suite and stay separate so a hint-text rewrite
# doesn't drift module-level RTL goldens.
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

  out=$(mktemp -t emitsvhint.XXXXXX); err=$(mktemp -t emitsvhint.XXXXXX)
  "$MATLABC" -emit-systemverilog "$m" >"$out" 2>"$err"; rc=$?
  ok=1
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $base: matlabc exited $rc (expected 0)"
    sed 's/^/  stderr: /' "$err"
    ok=0
  else
    # When .stderr is empty, assert stderr has no warning lines.
    if [[ ! -s "$exp" ]]; then
      if grep -qE "^.*warning:" "$err"; then
        echo "FAIL $base: expected no warnings, got:"
        grep -E "warning:" "$err" | sed 's/^/  /'
        ok=0
      fi
    else
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if ! grep -qF -- "$line" "$err"; then
          echo "FAIL $base: missing expected stderr substring"
          echo "  expected: $line"
          echo "  actual stderr:"
          sed 's/^/    /' "$err"
          ok=0
          break
        fi
      done < "$exp"
    fi
  fi
  rm -f "$out" "$err"
  if [[ $ok -eq 1 ]]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done

echo "----"
echo "emit-sv-hint passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
