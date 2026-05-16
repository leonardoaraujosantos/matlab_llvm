#!/usr/bin/env bash
# Golden-file test runner for the .mflow loader (Phase 1 of the
# flowchart frontend; see docs/flowchart_frontend.md).
#
# For each *.mflow file under this directory, runs `matlabc -dump-flow`
# and diffs the combined stdout+stderr against the matching *.expected
# file. Files under Errors/ are expected to fail (non-zero exit); files
# at the top level are expected to load cleanly (warnings allowed).
#
# Usage: run_tests.sh <path-to-matlabc>
#        UPDATE=1 run_tests.sh <path-to-matlabc>   # refresh goldens
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TESTDIR"

pass=0
fail=0
failed_names=()

run_one() {
  local f="$1" mode="$2" suffix="$3" flag="$4"
  local exp="${f%.mflow}${suffix}"
  # Replace the absolute path in diagnostics with a stable placeholder
  # so the goldens are portable across checkouts.
  local got
  got="$("$MATLABC" "$flag" "$f" 2>&1 | sed "s|$TESTDIR/||g")" || true
  if [[ -n "${UPDATE:-}" || ! -e "$exp" ]]; then
    printf '%s\n' "$got" > "$exp"
    echo "UPDATED ($mode) $f"
    return
  fi
  if diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    failed_names+=("$f ($mode)")
    echo "FAIL ($mode) $f"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
}

shopt -s nullglob
# FlowDoc-level dump: covers every .mflow regardless of dialect.
for f in *.mflow Errors/*.mflow StateChart/*.mflow StateChart/Errors/*.mflow; do
  run_one "$f" flow .expected -dump-flow
done
# Chart-IR dump: only state-chart fixtures, only the ones that load
# cleanly (Errors/ fail at the loader stage so a chart-IR run would
# repeat the same diagnostic).
for f in StateChart/*.mflow; do
  run_one "$f" chart .ir.expected -dump-chart
done
# Lowered-MATLAB dump: every clean state-chart fixture, captured so
# the lowering itself stays byte-stable across refactors.
for f in StateChart/*.mflow; do
  run_one "$f" matlab .matlab.expected -emit-matlab
done
# Interpreter trace: drive -simulate on every clean state-chart
# fixture and lock the deterministic trace as a regression signal.
for f in StateChart/*.mflow; do
  run_one "$f" simulate .sim.expected -simulate
done

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
