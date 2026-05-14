#!/usr/bin/env bash
# Golden-file test runner for the mflowLink signal-flow lowering
# (Tier B of docs/mflow_link_roadmap.md).
#
# For each *.mflow file under this directory, runs
# `matlabc -simulate --dry-run` and diffs the combined stdout+stderr
# against the matching *.expected file. Files under Errors/ are
# expected to fail (non-zero exit) — a rejected algebraic loop, an
# unknown/reserved kind, a subsystem cycle; files at the top level are
# expected to lower cleanly.
#
# Usage: run_tests.sh <path-to-matlabc>
#        UPDATE=1 run_tests.sh <path-to-matlabc>   # refresh goldens
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
# Resolve to an absolute path before the cd below — a relative
# matlabc would silently break once we change directory.
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TESTDIR"

pass=0
fail=0

run_one() {
  local f="$1"
  local exp="${f%.mflow}.expected"
  local got
  got="$("$MATLABC" -simulate --dry-run "$f" 2>&1 | sed "s|$TESTDIR/||g")" || true
  if [[ -n "${UPDATE:-}" || ! -e "$exp" ]]; then
    printf '%s\n' "$got" > "$exp"
    echo "UPDATED $f"
    return
  fi
  if diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $f"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
}

shopt -s nullglob
for f in *.mflow Errors/*.mflow; do
  run_one "$f"
done

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
